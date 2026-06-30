import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SPAN_RE = re.compile(
    r'<span(?:\s+class="[^"]*")?>(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)

NEGATION_CUES = [
    " no ",
    " denies ",
    " denied ",
    " without ",
    " negative for ",
    " no history of ",
    " not have ",
    " does not have ",
    " did not have ",
]

FAMILY_CUES = [
    "mother",
    "father",
    "sister",
    "brother",
    "parent",
    "parents",
    "maternal",
    "paternal",
    "family history",
    "relative",
    "relatives",
]


def strip_span_tags(text: str) -> str:
    return SPAN_RE.sub(lambda m: m.group(1), text)


def truncate_extra_generation(text: str) -> str:
    return re.split(r"\n\s*Note\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def sentence_split_keep_punctuation(text: str):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def should_remove_spans(sentence: str) -> bool:
    plain = " " + strip_span_tags(sentence).lower() + " "
    has_negation = any(cue in plain for cue in NEGATION_CUES)
    has_family = any(cue in plain for cue in FAMILY_CUES)
    return has_negation or has_family


def clean_module1_output(text: str) -> str:
    text = truncate_extra_generation(text)
    cleaned_sentences = []
    for sent in sentence_split_keep_punctuation(text):
        if should_remove_spans(sent):
            sent = strip_span_tags(sent)
        cleaned_sentences.append(sent)
    return "\n".join(cleaned_sentences)


def extract_spans(annotated_text: str):
    return [m.group(1).strip() for m in SPAN_RE.finditer(annotated_text)]


def build_prompt(note: str) -> str:
    instruction = f"""
You are an expert clinical information extraction system.

Your task is to return the same clinical note, but with rare disease phenotype mentions in the patient wrapped in <span class="condition"> and </span> tags.

Rules:
- Return only the annotated clinical note.
- Do not define or explain any terms.
- Do not add new sentences.
- Do not tag negated findings.
- Do not tag family history or conditions in relatives.
- Do not tag diseases or symptoms that are not present in the patient.

Example input:
The patient has developmental delay and hypotonia. He has no seizures. His father has diabetes.

Example output:
The patient has <span class="condition">developmental delay</span> and <span class="condition">hypotonia</span>. He has no seizures. His father has diabetes.

Now annotate this clinical note:
{note}
"""
    return f"<s>[INST] {instruction.strip()} [/INST]"


def generate_annotation(model, tokenizer, note: str, device: str, max_new_tokens: int) -> tuple[str, str, list[str]]:
    prompt = build_prompt(note)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    cleaned = clean_module1_output(raw)
    spans = extract_spans(cleaned)
    return raw, cleaned, spans


def main():
    parser = argparse.ArgumentParser(description="Run RARE-PHENIX Module 1 phenotype extraction on a CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--id-column", default="patient_id", help="Column containing patient/note IDs.")
    parser.add_argument("--text-column", default="note_text", help="Column containing clinical note text.")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-chat-hf", help="Base Llama model ID.")
    parser.add_argument("--adapter", default="shyrcathy/rare-phenix-llama2-7b-raredis", help="RARE-PHENIX PEFT adapter ID.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default=None, help="Device override: mps, cuda, or cpu.")
    parser.add_argument("--include-raw-output", action="store_true", help="Include raw uncleaned model output in the output CSV.")
    parser.add_argument(
        "--module2-output",
        default=None,
        help="Optional long-format CSV for Module 2 compatibility with columns UID and Step1_Clean_Split.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.to(device)
    model.eval()

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.id_column not in reader.fieldnames:
        raise ValueError(f"ID column '{args.id_column}' not found. Available columns: {reader.fieldnames}")
    if args.text_column not in reader.fieldnames:
        raise ValueError(f"Text column '{args.text_column}' not found. Available columns: {reader.fieldnames}")

    output_fields = [
        args.id_column,
        "cleaned_annotated_note",
        "extracted_phenotypes",
    ]
    if args.include_raw_output:
        output_fields.insert(1, "raw_model_output")

    module2_rows = []
    seen_by_uid = {}

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            note_id = row[args.id_column]
            note = row[args.text_column]
            print(f"Processing {i}/{len(rows)}: {note_id}")

            raw, cleaned, spans = generate_annotation(
                model=model,
                tokenizer=tokenizer,
                note=note,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            output_row = {
                args.id_column: note_id,
                "cleaned_annotated_note": cleaned,
                "extracted_phenotypes": "|".join(spans),
            }
            if args.include_raw_output:
                output_row["raw_model_output"] = raw

            writer.writerow(output_row)

            # Module 2 compatibility: one deduplicated phenotype per row.
            seen = seen_by_uid.setdefault(note_id, set())
            for span in spans:
                span_clean = span.strip()
                span_key = span_clean.lower()
                if span_clean and span_key not in seen:
                    module2_rows.append({
                        "UID": note_id,
                        "Step1_Clean_Split": span_clean,
                    })
                    seen.add(span_key)

    print(f"Done. Wrote: {output_path}")

    if args.module2_output:
        module2_output_path = Path(args.module2_output)
        module2_output_path.parent.mkdir(parents=True, exist_ok=True)

        with module2_output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["UID", "Step1_Clean_Split"])
            writer.writeheader()
            writer.writerows(module2_rows)

        print(f"Done. Wrote Module 2-compatible file: {module2_output_path}")


if __name__ == "__main__":
    main()

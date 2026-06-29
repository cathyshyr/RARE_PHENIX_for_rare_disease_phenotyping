import argparse
import re
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


def main():
    parser = argparse.ArgumentParser(description="Smoke test RARE-PHENIX Module 1 HF adapter.")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base Llama model ID.",
    )
    parser.add_argument(
        "--adapter",
        default="shyrcathy/rare-phenix-llama2-7b-raredis",
        help="RARE-PHENIX PEFT adapter model ID.",
    )
    parser.add_argument(
        "--note",
        default=(
            "The patient has global developmental delay, hypotonia, microcephaly, "
            "feeding difficulties, and short stature. She has no history of seizures. "
            "Her mother has diabetes."
        ),
        help="Clinical note to annotate. Use synthetic/deidentified text for testing.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.to(device)
    model.eval()

    prompt = build_prompt(args.note)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    cleaned = clean_module1_output(raw)

    print("\n===== RAW MODEL OUTPUT =====\n")
    print(raw)

    print("\n===== CLEANED MODULE 1 OUTPUT =====\n")
    print(cleaned)


if __name__ == "__main__":
    main()

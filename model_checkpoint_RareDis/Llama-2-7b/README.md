# RARE-PHENIX RareDis Checkpoint

This folder contains the **RareDis-only PEFT/LoRA adapter checkpoint** for RARE-PHENIX.

This checkpoint was fine-tuned only on the public RareDis corpus. It does **not** include the RARE-PHENIX model version fine-tuned on data derived from the NIH Undiagnosed Diseases Network (UDN) or other controlled-access data.

## Files

- `adapter_model.safetensors`: LoRA adapter weights
- `adapter_config.json`: PEFT adapter configuration
- `tokenizer_config.json`: tokenizer configuration
- `tokenizer.model`: tokenizer model
- `tokenizer.json`: tokenizer JSON file
- `special_tokens_map.json`: special tokens mapping

## Base Model

This checkpoint is an adapter for:

```text
meta-llama/Llama-2-7b-chat-hf
```

The base model is **not** included in this repository. Users must obtain access to the base model separately.

## Usage

This repository contains adapter weights only. To use the model, load the base model and then apply the adapter.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-2-7b-chat-hf"
adapter_path = "path/to/final_checkpoint"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

## Training Data

This checkpoint was fine-tuned using the public RareDis corpus:

Martinez-deMiguel, C., Segura-Bedmar, I., Chacon-Solano, E. and Guerrero-Aspizua, S., 2022. The RareDis corpus: a corpus annotated with rare diseases, their signs and symptoms. *Journal of Biomedical Informatics*, 125, p.103961.

## Important Note

The full RARE-PHENIX checkpoints trained with UDN-derived patient phenotype data are not publicly released due to controlled-access data sharing policies and potential patient privacy concerns.

This RareDis-only checkpoint is provided to support reproducibility and academic comparison using a publicly shareable version of the model.

## Citation

If you use this checkpoint, please cite the RARE-PHENIX paper and the RareDis corpus.

```bibtex
@article{shyr2026artificial,
  title={An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models},
  author={Shyr, Cathy and Hu, Yan and Tinker, Rory J and Cassini, Thomas A and Byram, Kevin W and Hamid, Rizwan and Fabbri, Daniel V and Wright, Adam and Peterson, Josh F and Bastarache, Lisa and others},
  journal={arXiv preprint arXiv:2602.20324},
  year={2026}
}

@article{martinez2022raredis,
  title={The RareDis corpus: a corpus annotated with rare diseases, their signs and symptoms},
  author={Mart{\'\i}nez-deMiguel, Claudia and Segura-Bedmar, Isabel and Chac{\'o}n-Solano, Esteban and Guerrero-Aspizua, Sara},
  journal={Journal of biomedical informatics},
  volume={125},
  pages={103961},
  year={2022},
  publisher={Elsevier}
}
```

## Limitations

This checkpoint is intended for research use only. It should not be used as a standalone clinical diagnostic system or as a substitute for medical expertise.

Because this checkpoint was fine-tuned only on the RareDis corpus, its behavior and performance may differ from the full RARE-PHENIX model described in the paper.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

## Training Data

This checkpoint was fine-tuned using the public RareDis corpus:

Martinez-deMiguel, C., Segura-Bedmar, I., Chacon-Solano, E. and Guerrero-Aspizua, S., 2022. The RareDis corpus: a corpus annotated with rare diseases, their signs and symptoms. *Journal of Biomedical Informatics*, 125, p.103961.

## Important Note

The full RARE-PHENIX checkpoints trained with UDN-derived patient phenotype data are not publicly released due to controlled-access data sharing policies and potential patient privacy concerns.

This RareDis-only checkpoint is provided to support reproducibility and academic comparison using a publicly shareable version of the model.

## Citation

If you use this checkpoint, please cite the RARE-PHENIX paper and the RareDis corpus.

```bibtex
@article{shyr2026artificial,
  title={An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models},
  author={Shyr, Cathy and Hu, Yan and Tinker, Rory J and Cassini, Thomas A and Byram, Kevin W and Hamid, Rizwan and Fabbri, Daniel V and Wright, Adam and Peterson, Josh F and Bastarache, Lisa and others},
  journal={arXiv preprint arXiv:2602.20324},
  year={2026}
}

@article{martinez2022raredis,
  title={The RareDis corpus: a corpus annotated with rare diseases, their signs and symptoms},
  author={Mart{\'\i}nez-deMiguel, Claudia and Segura-Bedmar, Isabel and Chac{\'o}n-Solano, Esteban and Guerrero-Aspizua, Sara},
  journal={Journal of biomedical informatics},
  volume={125},
  pages={103961},
  year={2022},
  publisher={Elsevier}
}
```

## Limitations

This checkpoint is intended for research use only. It should not be used as a standalone clinical diagnostic system or as a substitute for medical expertise.

Because this checkpoint was fine-tuned only on the RareDis corpus, its behavior and performance may differ from the full RARE-PHENIX model described in the paper.

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig


# # Load from latest checkpoint
# model = DistilBertForSequenceClassification.from_pretrained("distilbert/checkpoint-8142")
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert")

# # Save final model and tokenizer
# model.save_pretrained("distilbert_final")
# tokenizer.save_pretrained("distilbert_final")

model_path = r"C:\Users\Rohit\OneDrive\Desktop\CS598Project\distilbert\checkpoint-8142"
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=28  
)
adapter_path = model_path
peft_config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base_model, adapter_path)

save_path = r"C:\Users\Rohit\OneDrive\Desktop\CS598Project\distilbert-final"

base_model.save_pretrained(save_path)

tokenizer.save_pretrained(save_path)

peft_config.save_pretrained(save_path)

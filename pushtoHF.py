from huggingface_hub import HfApi
api = HfApi()


save_path = r"C:\Users\Rohit\OneDrive\Desktop\CS598Project\distilbert-final"


api.upload_folder(
    folder_path=save_path,
    repo_id="SaiRohitMurali/distilbertmodel-598",
    repo_type="model"
)
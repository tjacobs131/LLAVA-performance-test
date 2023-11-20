import replicate
import os
import glob

# --- Setup

# Get api key from file
script_dir = os.path.dirname(os.path.abspath(__file__))
key_file_path = os.path.abspath(os.path.join(script_dir, "..", "..", "key.txt"))
with open(key_file_path, "r") as key_file:
    api_key = key_file.read()

os.environ["REPLICATE_API_TOKEN"] = str(api_key)

# --- Get data

prompts = ["You are an expert in plant disease detection on leaves, are there any diseases that can you detect in this image?, your options are: rust (mild), rust (extreme), healthy. If you are not sure about the disease, you can say that you are unsure, but you must give your best answer at the end anyway."]

# Get image paths
image_dir = os.path.abspath(os.path.join(script_dir, "..", "ImageData", "img"))
print(f"Image directory: {image_dir}")  # Debug line

image_paths = glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True)

print("Amount of requests: " + str(image_dir.__len__() * prompts.__len__()))

# --- Get outputs from the lvlm

for prompt in prompts:
    print("--- Prompt: " + prompt + " ---")
    for image_path in image_paths:
        print("\nImage: " + image_path)
        with open(image_path, "rb") as image_file:
            output = replicate.run(
                "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
                input={"prompt": prompt,
                "image": image_file}
            )

            for item in output:
                print(item, end="")
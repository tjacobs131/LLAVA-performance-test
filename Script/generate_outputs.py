import replicate
import os
import glob


# --- Setup ----


# Get api key from file
script_dir = os.path.dirname(os.path.abspath(__file__))
key_file_path = os.path.abspath(os.path.join(script_dir, "..", "..", "key.txt"))
with open(key_file_path, "r") as key_file:
    api_key = key_file.read()

os.environ["REPLICATE_API_TOKEN"] = str(api_key)


# --- Get data ---


prompts = ["You are an expert in plant disease detection on leaves, are there any diseases that can you detect in this image?, your options are: rust (mild), rust (extreme), healthy. You are a part of an automatic farming control loop, therefore a final answer is absolutely necessary. Your job is to reason about the health status about the leaf that is provided. Output your final answer exactly like this: Final answer: [your choice out of the given options]. It does not matter how sure you are about your answer, it is crucial that you make your best guess anyway."]

# Get images
image_dir = os.path.abspath(os.path.join(script_dir, "..", "ImageData", "img"))
image_paths = glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True)

print("Amount of requests: " + str(image_dir.__len__() * prompts.__len__()))
print("Estimated run time: " + str(len(image_dir) * len(prompts) * 5) + " sec.")


# --- Get outputs from the LVLM ---

outputs = []
image_count = 0

for prompt in prompts:
    print("--- Prompt: " + prompt + " ---")
    for image_path in image_paths:
        print("\n\nImage #" + str(image_count) +": "  + image_path)
        outputs.append("")
        with open(image_path, "rb") as image_file:
            output = replicate.run(
                "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
                input={"prompt": prompt,
                "image": image_file}
            )

            for item in output:
                outputs[image_count] += item
                print(item, end="")

            image_count += 1    

for output in outputs:
    print(output + "\n")


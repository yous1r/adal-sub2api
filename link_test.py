from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-sub2api-secret")

stream = client.chat.completions.create(
    model="openai-gpt-5.6-terra", messages=[{"role":"user","content":"count 1 2 3"}], stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
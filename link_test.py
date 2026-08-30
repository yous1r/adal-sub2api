from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:48080/v1", api_key="sk-sub2api-secret")

r = client.chat.completions.create(
    model="openai-gpt-5.6-luna",
    messages=[{"role": "user", "content": "hi"}],
)

pass

# stream = client.chat.completions.create(
#     model="openai-gpt-5.6-terra", messages=[{"role":"user","content":"count 1 2 3"}], stream=True)
# for chunk in stream:
#     print(chunk.choices[0].delta.content or "", end="", flush=True)
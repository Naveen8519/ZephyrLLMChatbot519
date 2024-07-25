import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "I'm your virtual shopping assistant, here to help you find the perfect items and answer any questions you may have. Whether you're looking for the latest fashion trends, tech gadgets, or home essentials, I'm here to make your shopping experience as smooth and enjoyable as possible."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "I'm your virtual shopping assistant, here to help you find the perfect items and answer any questions you may have. Whether you're looking for the latest fashion trends, tech gadgets, or home essentials, I'm here to make your shopping experience as smooth and enjoyable as possible.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["Can you compare the features of the iPhone 14 and the Samsung Galaxy S23?"],
        ["I'm looking for a birthday gift for my mom. Can you suggest something special under $100?"],
        ["I need help setting up my new smart TV. Can you provide setup instructions or support?"]
    ],
    title = 'Virtual ShopMate🛍️'
)


if __name__ == "__main__":
    demo.launch()
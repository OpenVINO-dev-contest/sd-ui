from optimum.intel.openvino import OVStableDiffusionPipeline
from pathlib import Path
from openvino.runtime import Core
import gradio as gr
import time

core = Core()
available_devices = core.available_devices
current_device = "CPU"

available_language = ["English", "中文"]
current_language = "English"

model_path = Path('../ir_model')
cn_model_path = Path('../ir_model_cn')

example_text = ["red car in snowy forest", "机械狗"]
example_negative_text = ["low quality, ugly, deformed, blur", "不清晰的"]
available_model_path = [model_path, cn_model_path]

model_language_dict = dict(zip(available_language, available_model_path))
text_dict = dict(zip(available_language, example_text))
negative_text_dict = dict(zip(available_language, example_negative_text))

ov_pipe = OVStableDiffusionPipeline.from_pretrained(available_model_path[0],
                                                    device=current_device,
                                                    compile=False)
ov_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
ov_pipe.compile()


def generate_from_text(text,
                       negative_text,
                       num_steps,
                       scale,
                       _=gr.Progress(track_tqdm=True)):
    start = time.perf_counter()
    output = ov_pipe(prompt=text,
                     negative_prompt=negative_text,
                     num_inference_steps=num_steps,
                     guidance_scale=scale).images[0]
    end = time.perf_counter()
    perf = "Time cost: {:.3f}s".format(end - start)
    return output, perf


def select_device(device_str: str,
                 current_text: str,
                 current_negative_text: str,
                  progress: gr.Progress = gr.Progress()):
    if device_str != ov_pipe._device:
        ov_pipe.to(device_str)

        for i in progress.tqdm(range(1),
                               desc=f"Model loading on {device_str}"):
            ov_pipe.compile()
    return current_text, current_negative_text


def select_model(language_str: str,
                 device_str: str,
                 current_text: str,
                 current_negative_text: str,
                 progress: gr.Progress = gr.Progress()):
    global current_language
    global ov_pipe
    if language_str != current_language:
        current_text = text_dict[language_str]
        current_negative_text = negative_text_dict[language_str]
        ov_pipe = OVStableDiffusionPipeline.from_pretrained(
            model_language_dict[language_str],
            device=device_str,
            compile=False)
        ov_pipe.reshape(batch_size=1,
                        height=512,
                        width=512,
                        num_images_per_prompt=1)
        for i in progress.tqdm(range(1), desc=f"Loading {language_str} model"):
            ov_pipe.compile()
            current_language = language_str
    return current_text, current_negative_text


def reset(response, perf):
    return None, ""


examples = [
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    "城堡 大海 夕阳 宫崎骏动画",
    "花落知多少",
    "小桥流水人家",
    "飞流直下三千尺，油画",
    "中国海边城市，科幻，未来感，唯美，插画。"
]

with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion using OpenVINO.\n")

    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(value=example_text[0], label="Enter your prompt")
            negative_text = gr.Textbox(value=example_negative_text[0],
                                       label="Enter a negative prompt")
            model_output = gr.Image(label="Result", type="pil", height=512)
            performance = gr.Textbox(label="Performance",
                                     lines=1,
                                     interactive=False)
            with gr.Row(scale=1):
                button_submit = gr.Button(value='Submit', variant='primary')
                button_clear = gr.Button(value="Clear")

        with gr.Column(scale=1):
            language = gr.Dropdown(choices=available_language,
                                   value=current_language,
                                   label="Model language")
            device = gr.Dropdown(choices=available_devices,
                                 value=current_device,
                                 label="Device")
            num_steps = gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                interactive=True,
                label="Number of steps",
            )
            guidance_scale = gr.Slider(
                minimum=0.5,
                maximum=30.0,
                value=7.5,
                step=0.5,
                interactive=True,
                label="Guidance scale",
            )
            gr.Examples(examples, text)
    button_submit.click(generate_from_text,
                        [text, negative_text, num_steps, guidance_scale],
                        [model_output, performance])
    button_clear.click(reset, [model_output, performance],
                       [model_output, performance])
    device.change(select_device, [device, text, negative_text], [text, negative_text])
    language.change(select_model, [language, device, text, negative_text],
                    [text, negative_text])

if __name__ == "__main__":
    try:
        demo.queue().launch(debug=True,
                            share=False,
                            height=800)
    except Exception:
        demo.queue().launch(debug=True, share=True, height=800)
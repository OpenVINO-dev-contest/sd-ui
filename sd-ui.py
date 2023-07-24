from optimum.intel.openvino import OVStableDiffusionPipeline
from pathlib import Path
from openvino.runtime import Core
import gradio as gr
import time

core = Core()
available_devices = core.available_devices
current_device = "CPU"

model_id = "stabilityai/stable-diffusion-2-1-base"

model_path = Path('ir_model')

if model_path.exists():
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_path,
                                                        device=current_device,
                                                        compile=False)
    ov_pipe.reshape(batch_size=1,
                    height=512,
                    width=512,
                    num_images_per_prompt=1)
else:
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id,
                                                        device=current_device,
                                                        export=True,
                                                        compile=False)
    ov_pipe.reshape(batch_size=1,
                    height=512,
                    width=512,
                    num_images_per_prompt=1)
    ov_pipe.save_pretrained(model_path)
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


def reset(response, perf):
    return None, ""


examples = [
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation"
]

with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion using OpenVINO.\n")

    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(value="red car in snowy forest",
                              label="Enter your prompt")
            negative_text = gr.Textbox(
                value="low quality, ugly, deformed, blur",
                label="Enter a negative prompt")
            model_output = gr.Image(label="Result", type="pil", height=512)
            performance = gr.Textbox(label="Performance",
                                     lines=1,
                                     interactive=False)
            with gr.Column(scale=1):
                button_submit = gr.Button(value="Generate image")
                button_clear = gr.Button(value="Clear")

        with gr.Column(scale=1):
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

if __name__ == "__main__":
    try:
        demo.queue().launch(debug=True,
                            share=False,
                            height=800)
    except Exception:
        demo.queue().launch(debug=True, share=True, height=800)

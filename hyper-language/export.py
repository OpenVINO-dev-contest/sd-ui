from optimum.intel.openvino import OVStableDiffusionPipeline
from pathlib import Path

model_id = "stabilityai/stable-diffusion-2-1-base"
cn_model_id = "SkyWork/SkyPaint"

model_path = Path('../ir_model')
cn_model_path = Path('../ir_model_cn')

if model_path.exists()==False:
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id,
                                                        export=True,
                                                        compile=False)
    ov_pipe.reshape(batch_size=1,
                    height=512,
                    width=512,
                    num_images_per_prompt=1)
    ov_pipe.save_pretrained(model_path)

if cn_model_path.exists()==False:
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(cn_model_id,
                                                        export=True,
                                                        compile=False)
    ov_pipe.reshape(batch_size=1,
                    height=512,
                    width=512,
                    num_images_per_prompt=1)
    ov_pipe.save_pretrained(cn_model_path)
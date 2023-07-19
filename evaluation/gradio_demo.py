import gradio  as gr
import numpy as np
import sys, os
from basic_ui import BasicUI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from confignet import ConfigNet

# global is used to store the model so we don't have to reload it every time but it's not a good practice update it
confignet_model = ConfigNet.load(
    "/home2/xcnf86/confignet_stylegan/ConfigNetLight2D/experiments/celeba_focal_single_discriminator/checkpoints/final.json"
)


basic_ui = BasicUI(confignet_model)

def get_new_embeddings(input_img: gr.Image):
    embedding = confignet_model.encode_images(input_img)
    return embedding, input_img


def get_embedding_with_new_attribute_value(
    parameter_name: str, input_img: gr.Image
) -> np.ndarray:
    
    input_img = input_img[:,:,::-1]
    """Samples a new value of the currently controlled face attribute and sets in the latent embedding"""
    new_param_value = confignet_model.facemodel_param_distributions[
        parameter_name
    ].sample(1)[0]
    latents=confignet_model.encode_images(input_img[np.newaxis])
    modified_latents = confignet_model.set_facemodel_param_in_latents(
        latents, parameter_name, new_param_value
    )
    generated=confignet_model.generate_images(modified_latents)

    return generated[0][:,:,::-1]


def fine_tune(input_img: gr.Image):
    input_img = input_img[:,:,::-1]
    n_fine_tuning_iters = 100
    print("Fine tuning generator on single image, this might take a minute or two")
    (current_embedding_unmodified,) = confignet_model.fine_tune_on_img(
        input_img, n_fine_tuning_iters
    )

    generated=confignet_model.generate_images(current_embedding_unmodified[np.newaxis])
 
    return generated[0][:,:,::-1]
 

def reconstruct(input_img: gr.Image):
    input_img = input_img[:,:,::-1]
    # Set next embedding value for rendering
    current_renderer_input = confignet_model.encode_images(input_img[np.newaxis])

    basic_ui.set_next_embeddings(current_renderer_input)
    generated=confignet_model.generate_images(current_renderer_input)
    return generated[0][:,:,::-1]


with gr.Blocks(theme='monochrome') as demo:
    gr.Markdown("""
                <img src='https://durham.ac.uk/media/durham-university/site-assets/image/logo-dark.svg' height="100" width="100" style="float:right">
                
                
                # Controllable Racial Phenotype Demo 2023
                 
                """, style="font-size: 20px; font-weight: bold; text-align: center; margin-bottom: 20px; margin-top: 20px; color: #000000; font-family: 'Open Sans', sans-serif;")
    with gr.Row():
        # Sample latent embeddings from input images if available and if not sample from Latent GAN
        input_img = gr.Image(shape=(256, 256),height=256,width=256,label="Original Image")
        output_img = gr.Image(shape=(256, 256),height=256,width=256,label="Generated Image")
    
    reconstruct_buttton = gr.Button("Reconstruct",size='lg')
    reconstruct_buttton.click(reconstruct,inputs=input_img,outputs=output_img,api_name="reconstruct")

    finetune_btn = gr.Button("Fine Tune")
    finetune_btn.click(
        fn=fine_tune,
        inputs=input_img,
        outputs=output_img,
        api_name="finetune",
    )

    control_skin_button = gr.Button("Control Skin Color")
    control_skin_button.click(
        get_embedding_with_new_attribute_value,
        inputs=[gr.Text('skin_color',visible=False),input_img],
        outputs=output_img,
        api_name="skin",
    )
    control_hair_button = gr.Button("Control Hair Color")
    control_hair_button.click(
        get_embedding_with_new_attribute_value,
        inputs=[gr.Text('hair_color',visible=False),input_img],
        outputs=output_img,
        api_name="hair",
    )


demo.launch(share=False, debug=True,server_name="0.0.0.0", server_port=7878)

# Copyright (c) 2021 Justin Pinkney

import runway
from runway.data_types import category, vector, image, number

import editor
import face_detection

edit_controls = {k: number(description=k, default=0, min=-20, max=20) for k in editor.edits.keys()}
inputs = {'original': image()}
inputs.update(edit_controls)
outputs = { 'image': image() }


@runway.setup(options={
        'checkpoint': runway.file(extension='.pt', default="psp_ffhq_encode.pt"),
        'face_detector': runway.file(extension='.dat', default="shape_predictor_5_face_landmarks.dat"),
    })
def setup(opts):
    checkpoint_path = opts['checkpoint']
    face_detection.MODEL_PATH = opts['face_detector']

    encoder, decoder, latent_avg = editor.load_model(checkpoint_path)

    manipulator = editor.manipulate_model(decoder)
    manipulator.edits = {editor.idx_dict[v[0]]: {v[1]: 0} for k, v in editor.edits.items()}

    return encoder, decoder, latent_avg, manipulator


@runway.command('encode', inputs=inputs, outputs=outputs, description='Generate an image.')
def generate(model, input_args):
    
    encoder, decoder, latent_avg, manipulator = model

    original = input_args['original']
    input_size = [1024, 1024]
    
    output = original.copy()
    cropped, n_faces, quad = face_detection.align(original)

    for k, v in editor.edits.items():
        layer_index, channel_index, sense = v
        conv_name = editor.idx_dict[layer_index]
        manipulator.edits[conv_name][channel_index] = input_args[k]*sense
    
    for i in range(n_faces):
        # We already did the first one above
        if i > 0:
            cropped, _, quad = face_detection.align(original, face_index=i)

        transformed_crop = editor.run(encoder, decoder, latent_avg, cropped)
    
        output = face_detection.composite_images(quad, transformed_crop, output)

    return output


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=9000)
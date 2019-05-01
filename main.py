
# returns the feature tensors of each image, as they are forwarded through the trained model
# assumes that the images are already in Tensor form
def get_features(imgA, imgB, model):
    F_A = model.forward(imgA).data
    F_B = model.forward(imgB).data
    return [F_A, F_B]

# R5 = (F_A, F_B) --> the entire domain of F_A and F_B
# Then, go through other layers and...
# TODO: get P and Q from the feature tensors
# TODO: get p's and q's from each subset of P and Q
import numpy as np
import onnxruntime as rt

class ONNXModelInference():
    def __init__(self, model):
        self.sess = rt.InferenceSession(model.SerializeToString())

    def predict(self, X):
        input_name = self.sess.get_inputs()[0].name

        outputs = self.sess.run(None, {input_name: X.values.astype(np.float32)})
        return np.array([d[1] for d in outputs[1]])

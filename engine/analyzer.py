
import cv2
import numpy as np
import innvestigate
import innvestigate.utils as iutils


class Analyzer:
    """ Interpretability visualization with saliency maps """

    def __init__(self):
        pass

    def load_images(self, paths, img_w=224, img_h=224):
        # paths = glob(path)
        # print('++ paths', paths)
        for p in paths:
            img = cv2.imread(p)
            img_resize = cv2.resize(img, dsize=(img_w, img_h), interpolation=cv2.INTER_CUBIC)
            yield np.expand_dims(img_resize, axis=0)

    def deepTaylorAnalyzer(self, img_expand, model):
        analyzer = innvestigate.create_analyzer("deep_taylor", model)
        analysis = analyzer.analyze(img_expand)

        ## Aggregate along color channels and normalize to [-1, 1]
        a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
        a /= np.max(np.abs(a))
        return a

    def deepTaylor_SoftAnalyzer(self, img_expand, model):

        ## Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
        # model_wo_sm = iutils.keras.graph.get_model_execution_graph(model, keep_input_layers=False)

        ## Creating an analyzer
        analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_sm)

        ## Applying the analyzer
        analysis = analyzer.analyze(img_expand)

        ## Aggregate along color channels and normalize to [-1, 1]
        # a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
        # a /= np.max(np.abs(a))

        # Handle input depending on model and backend.
        channels_first = keras.backend.image_data_format() == "channels_first"
        color_conversion = "BGRtoRGB"

        # Apply common postprocessing, e.g., re-ordering the channels for plotting.
        a = imgnetutils.postprocess(analysis, color_conversion, channels_first)
        a = imgnetutils.heatmap(a)
        return a

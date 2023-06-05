import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';

async function loadModel() {
  const model = await cocoSsd.load();
  return model;
}

async function detectObjects(model, image) {
  const predictions = await model.detect(image);
  return predictions;
}

async function run() {
  const model = await loadModel();

  const imageElement = document.getElementById('image');
  const image = tf.browser.fromPixels(imageElement);

  const predictions = await detectObjects(model, image);
  console.log(predictions);
}

run();

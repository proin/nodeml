'use strict';

const {evaluate, CNN, sample} = require('../index');

const bulk = sample.yeast();

let cnn = new CNN();
cnn.configure({learning_rate: 0.1, momentum: 0.001, batch_size: 1, l2_decay: 0.0001});

let layer = [];
layer.push({type: 'input', out_sx: 1, out_sy: 1, out_depth: 4});
layer.push({type: 'svm', num_classes: 10});
cnn.makeLayer(layer);

cnn.train(bulk.dataset, bulk.labels);
let result = cnn.test(bulk.dataset);
let evaluation = evaluate.accuracy(bulk.labels, result);
console.log(evaluation.micro.PRECISION);

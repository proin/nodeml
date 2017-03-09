'use strict';

const {Bayes, CNN, sample} = require('../index');

const yeast = sample.yeast();

// let map = {};
// let mapCnt = 0;
// for (let i = 0; i < yeast.labels.length; i++) {
//     if (!map[yeast.labels[i]] && map[yeast.labels[i]] !== 0) {
//         map[yeast.labels[i]] = mapCnt;
//         mapCnt++;
//     }
//     yeast.labels[i] = map[yeast.labels[i]];
// }


// Class: cnn
let cnn = new CNN();
cnn.configure({learning_rate: 0.1, momentum: 0.001, batch_size: 5, l2_decay: 0.0001});

var layer = [];
layer.push({type: 'input', out_sx: 1, out_sy: 1, out_depth: 8});
layer.push({type: 'svm', num_classes: 10});

cnn.setModel(layer);

for (let j = 0; j < 10; j++)
    for (let i = 0; i < yeast.dataset.length; i++)
        cnn.train(yeast.dataset[i], yeast.labels[i]);

let testCnt = 0;
for (let j = 0; j < yeast.dataset.length; j++) {
    let test = cnn.test(yeast.dataset[j]);
    if (test.answer == yeast.labels[j])
        testCnt++;
}

console.log(`svm: ${Math.round(testCnt * 10000 / yeast.dataset.length) / 100}%`);

// Class: bayes
let bayes = new Bayes();
bayes.train(yeast.dataset, yeast.labels);
testCnt = 0;
for (let j = 0; j < yeast.dataset.length; j++) {
    let test = bayes.test(yeast.dataset[j]);
    if (test.answer == yeast.labels[j])
        testCnt++;
}

console.log(`bayes: ${Math.round(testCnt * 10000 / yeast.dataset.length) / 100}%`);
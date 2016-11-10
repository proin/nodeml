'use strict';

const fs = require('fs');

const {bayes, sample} = require('./index');

let bbcClassifier = bayes();
sample.bbc()
    .then((sample)=> bbcClassifier.train(sample.dataset, sample.labels))
    .then(()=> sample.bbc())
    .then((sample)=> bbcClassifier.test(sample.dataset, {labels: sample.labels}))
    .catch((e)=> {
        console.log(e);
    })
    .then((result)=> {
        console.log(result.precision);
    });

return;

let dataset = [];

dataset.push({'fun': 1, 'couple': 1});
dataset.push({'fast': 1, 'furious': 1, 'shoot': 1});
dataset.push({'couple': 1, 'fast': 1, 'fun': 1});
dataset.push({'furious': 1, 'shoot': 1, 'fun': 1});
dataset.push({'fly': 1, 'fast': 1, 'shoot': 1, 'love': 1});

let labels = [];
labels.push('comedy');
labels.push('action');
labels.push('comedy');
labels.push('action');
labels.push('action');
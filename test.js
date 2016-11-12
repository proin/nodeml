'use strict';

const fs = require('fs');

const {ml, sample} = require('./index');
const {Bayes} = ml;

let bayes = new Bayes();
sample.bbc()
    .then((sample)=> bayes.train(sample.dataset, sample.labels))
    .then(()=> sample.bbc())
    .then((sample)=> bayes.test(sample.dataset, {labels: sample.labels}))
    .catch((e)=> {
        console.log(e);
    })
    .then((result)=> {
        console.log(result.precision);
    });

let bayes2 = new Bayes();
let dataset = [];

dataset.push({'fun': 3, 'couple': 1});
dataset.push({'couple': 1, 'fast': 1, 'fun': 3});
dataset.push({'fast': 3, 'furious': 2, 'shoot': 2});
dataset.push({'furious': 2, 'shoot': 4, 'fun': 1});
dataset.push({'fly': 2, 'fast': 3, 'shoot': 2, 'love': 1});

let labels = [];
labels.push('comedy');
labels.push('comedy');
labels.push('action');
labels.push('action');
labels.push('action');

bayes2.train(dataset, labels)
    .then(()=> bayes2.test([{'fun': 3, 'fast': 3, 'shoot': 2}, {'fun': 8, 'fast': 2, 'shoot': 2}]))
    .then((result)=> console.log(result));
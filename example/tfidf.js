'use strict';

const {feature, evaluate, Bayes, sample} = require('../index');

let bayes = new Bayes();
const bulk = sample.bbc();

// calculate tfidf
bulk.dataset = feature.tfidf(bulk.dataset, 'replace');

bayes.train(bulk.dataset, bulk.labels);
let result = bayes.test(bulk.dataset, {score: false});
let evaluation = evaluate.accuracy(bulk.labels, result);

console.log(evaluation.micro.PRECISION);

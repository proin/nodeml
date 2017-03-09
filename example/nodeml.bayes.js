'use strict';

const {evaluate, Bayes, sample} = require('../index');

// train data and test

let bayes = new Bayes();

bayes.train({'fun': 3, 'couple': 1}, 'comedy');
bayes.train({'couple': 1, 'fast': 1, 'fun': 3}, 'comedy');
bayes.train({'fast': 3, 'furious': 2, 'shoot': 2}, 'action');
bayes.train({'furious': 2, 'shoot': 4, 'fun': 1}, 'action');
bayes.train({'fly': 2, 'fast': 3, 'shoot': 2, 'love': 1}, 'action');

let result = bayes.test({'fun': 3, 'fast': 3, 'shoot': 2}, true);
console.log(result);

// train bulk data and evaluate result

bayes = new Bayes();

const bbc = sample.bbc();

bayes.train(bbc.dataset, bbc.labels);
result = bayes.test(bbc.dataset);
let evaluation = evaluate.accuracy(bbc.labels, result);

console.log(evaluation);
'use strict';

const {evaluate, Bayes, sample} = require('../index');

// train data and test

let bayes = new Bayes();

bayes.train({'fun': 3, 'couple': 1}, 'comedy');
bayes.train({'couple': 1, 'fast': 1, 'fun': 3}, 'comedy');
bayes.train({'fast': 3, 'furious': 2, 'shoot': 2}, 'action');
bayes.train({'furious': 2, 'shoot': 4, 'fun': 1}, 'action');
bayes.train({'fly': 2, 'fast': 3, 'shoot': 2, 'love': 1}, 'action');

let result = bayes.test({'fun': 3, 'fast': 3, 'shoot': 2}, {score: true});
console.log(result);

// train bulk data and evaluate result

bayes = new Bayes();

const bulk = sample.iris();

bayes.train(bulk.dataset, bulk.labels);
result = bayes.test(bulk.dataset, {score: true});

for (let j = 0; j < result.length; j++) {
    let res = [];
    for (let key in result[j].score)
        res.push({cls: key, score: result[j].score[key]});
    res.sort((a, b) => b.score - a.score);
    res = res.splice(0, 3);
    console.log(res);
    return;
}

let evaluation = evaluate.accuracy(bulk.labels, result);

console.log(evaluation.micro.PRECISION, evaluation.macro.PRECISION);

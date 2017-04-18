'use strict';

const {evaluate, kMeans, sample} = require('../index');

const dataset = sample.iris();

// unsupervised learning
let kmeans = new kMeans();
kmeans.train(dataset.dataset, { 
	k: 3 , dm: 0.001, iter: 20,
	proc: (iter, j, d)=> console.log(`iteration #${iter}: ${d}`)
});
kmeans.test(dataset.dataset);

// supervised learning
kmeans = new kMeans();
kmeans.train(dataset.dataset, { 
	labels: dataset.labels  
});
let result = kmeans.test(dataset.dataset);

let evaluation = evaluate.accuracy(dataset.labels, result);
console.log(evaluation.micro.PRECISION);

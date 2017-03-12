'use strict';

const {sample, CF, evaluation} = require('../index');

// load movie dataset: user_id movie_id rating like
const movie = sample.movie();

// split data as train & test
let train = [], test = [];
for (let i = 0; i < movie.length; i++) {
    if (Math.random() > 0.8) test.push(movie[i]);
    else train.push(movie[i]);
}

// set data
const cf = new CF();

cf.maxRelatedItem = 50;
cf.maxRelatedUser = 50;

cf.train(train, 'user_id', 'movie_id', 'rating');

// select 100 data for recommendation
let gt = cf.gt(test, 'user_id', 'movie_id', 'rating');
let gtr = {};
let users = [];
for (let user in gt) {
    gtr[user] = gt[user];
    users.push(user);
    if (users.length === 100) break;
}

// recommend for 100 users, each 40 movie
let result = cf.recommendToUsers(users, 40);

// evaluate by ndcg
let ndcg = evaluation.ndcg(gtr, result);
console.log(ndcg);
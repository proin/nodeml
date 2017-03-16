'use strict';

((app) => {
    app.sample = require('./libs/sample');

    // classification
    app.bayes = app.Bayes = require('./libs/bayes');
    app.knn = app.kNN = require('./libs/knn');
    app.cnn = app.CNN = require('./libs/cnn');

    // recommendation
    app.collaborativeFiltering = app.cf = app.CF = require('./libs/cf');

    // evaluate
    app.eval = app.evaluate = app.evaluation = require('./libs/evaluation/index');
})(module.exports);
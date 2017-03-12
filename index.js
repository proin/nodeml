'use strict';

((app)=> {
    app.sample = require('./libs/sample');

    // classification
    app.bayes = app.Bayes = require('./libs/bayes/index');
    app.knn = app.kNN= require('./libs/knn/index');
    app.cnn = app.CNN = require('./libs/cnn/index');

    // recommendation
    app.cf = app.CF = require('./libs/cf/index');

    // evaluate
    app.eval = app.evaluate = app.evaluation = require('./libs/evaluation/index');
})(module.exports);
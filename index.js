'use strict';

((app)=> {
    app.sample = require('./libs/sample');

    app.bayes = app.Bayes = require('./libs/bayes/index');
    app.cnn = app.CNN = require('./libs/cnn/index');
    app.cf = app.CF = require('./libs/cf/index');

    app.eval = app.evaluate = app.evaluation = require('./libs/evaluation/index');
})(module.exports);
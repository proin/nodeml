'use strict';

((app)=> {
    app.sample = require('./libs/sample');
    app.Bayes = require('./libs/bayes/index');
    app.CNN = require('./libs/cnn/index');
    app.eval = app.evaluate = app.evaluation = require('./libs/evaluation/index');
})(module.exports);
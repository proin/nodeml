'use strict';

((app)=> {
    app.sample = require('./libs/sample');

    app.ml = {};
    app.ml.Bayes = require('./libs/ml/bayes');
})(module.exports);
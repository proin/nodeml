'use strict';

((app)=> {
    app.sample = require('./libs/sample');
    app.Bayes = require('./libs/bayes/index');
    app.CNN = require('./libs/cnn/index');
})(module.exports);
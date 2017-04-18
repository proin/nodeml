'use strict';

((app) => {
	// sample
    app.sample = require('./libs/sample');

	// feature selection
	app.feature = {};
	app.feature.tfidf = require('./libs/feature-selection/tfidf');

    // classification
    app.bayes = app.Bayes = require('./libs/bayes');
    app.knn = app.kNN = require('./libs/knn');
    app.cnn = app.CNN = require('./libs/cnn');
	app.logistic = app.Logistic = require('./libs/logistic');
	app.kMeans = app.kmeans = require('./libs/kmeans');

    // recommendation
    app.collaborativeFiltering = app.cf = app.CF = require('./libs/cf');

    // evaluate
    app.eval = app.evaluate = app.evaluation = require('./libs/evaluation/index');
})(module.exports);

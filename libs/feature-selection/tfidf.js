'use strict';

module.exports = (dataset, type)=> {
	let idf = {};

	for (let i = 0; i < dataset.length; i++) {
		if (!dataset[i]) continue;
		let features = dataset[i];
		for (let feature in features) {
			if (!idf[feature]) idf[feature] = 0;
			idf[feature]++;
		}
	}

	for (let feature in idf)
		idf[feature] = Math.log(dataset.length / idf[feature]);

	let result = [];
	for (let i = 0; i < dataset.length; i++) {
		if (!dataset[i]) continue;

		let features = dataset[i];
		let max = 0;
		for (let feature in features)
			if (features[feature] > max)
				max = features[feature];

		for (let feature in features) {
			let tf = (0.5 + 0.5 * features[feature]) / max;
			let tfidf = tf * (idf[feature] ? idf[feature] : 0);
			if(type === 'replace') 
				features[feature] = tfidf;
			else
				features[feature] = features[feature] * tfidf;
		}

		result[i] = features;
	}

	return result;
};

const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options) {
        this.labels = labels
        this.features = this.processFeatures(features)
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options)
        this.weigths = tf.zeros([2, 1])
    }

    gradientDescent() {
        const currentGuesses = this.features.matMul(this.weigths)
        const differences = currentGuesses.sub(this.labels)
        const slopes = this.features
            .transpose()
            .matMul(differences)
            .div(this.features.shape[0])
        this.weigths = this.weigths.sub(slopes.mul(this.options.learningRate))
    }

    // gradientDescent() {
    //     const currentGuessesForMPG = this.features.map(row => {
    //         return this.m * row[0] + this.b
    //     })
    //     const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return guess - this.labels[i][0]
    //     })) * 2 / this.features.length

    //     const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return -1 * this.features[i][0] * (this.labels[i][0] - guess)
    //     })) * 2 / this.features.length
    //     this.b = this.b - bSlope * this.options.learningRate
    //     this.m = this.m - mSlope * this.options.learningRate
    // }

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent()
        }
    }

    test(testFeatures, testLabels) {
        testLabels = tf.tensor(testLabels)
        testFeatures = this.processFeatures(testFeatures)
        const predictions = testFeatures.matMul(this.weigths)
        const res = testLabels.sub(predictions).pow(2).sum().get()
        const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get()
        return 1 - res / tot
    }

    processFeatures(features) {
        features = tf.tensor(features)
        features = tf.ones([features.shape[0], 1]).concat(features, 1)
        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5)) // we have to reuse mean and variance for the further standardization (test set)
        } else {
            features = this.standardize(features) // running standardization for the first time
        }
        return features
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0)
        this.mean = mean
        this.variance = variance
        return features.sub(mean).div((variance).pow(0.5))
    }
}

module.exports = LinearRegression
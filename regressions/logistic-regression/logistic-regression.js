const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LogisticRegression {
    constructor(features, labels, options) {
        this.labels = tf.tensor(labels)
        this.features = this.processFeatures(features)
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options)
        this.weigths = tf.zeros([this.features.shape[1], 1])
        this.mseHistory = []
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weigths).sigmoid()
        const differences = currentGuesses.sub(labels)
        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])
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
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize)
        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) {
                const startIndex = j * this.options.batchSize
                const { batchSize } = this.options
                const featuresSlice = this.features.slice([startIndex, 0], [batchSize, -1])
                const labelsSlice = this.labels.slice([startIndex, 0], [batchSize, -1])
                this.gradientDescent(featuresSlice, labelsSlice)
            }
            this.recordMSE()
            this.updateLearningRate()
        }
    }

    predict(observations) {
        return this.processFeatures(observations).matMul(this.weigths).sigmoid()
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures).round()
        testLabels = tf.tensor(testLabels)
        const incorrect = predictions.sub(testLabels).abs().sum()
        return (predictions.shape[0] - incorrect.get()) / predictions.shape[0]
    }

    processFeatures(features) {
        features = tf.tensor(features)

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5)) // we have to reuse mean and variance for the further standardization (test set)
        } else {
            features = this.standardize(features) // running standardization for the first time
        }
        features = tf.ones([features.shape[0], 1]).concat(features, 1)
        return features
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0)
        this.mean = mean
        this.variance = variance
        return features.sub(mean).div((variance).pow(0.5))
    }

    recordMSE() {
        const mse = this.features.
            matMul(this.weigths)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get()
        this.mseHistory.unshift(mse) // add to the beginning instead of the end
    }

    updateLearningRate() {
        if (this.mseHistory.length < 2) {
            return
        }
        const lastValue = this.mseHistory[0] // newer records are in the beginning
        const secondLast = this.mseHistory[1]
        if (lastValue > secondLast) {
            this.options.learningRate /= 2
        } else {
            this.options.learningRate *= 1.05
        }
    }
}

module.exports = LogisticRegression
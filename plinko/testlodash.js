const _  = require('lodash')
const data = [[50,1],[25,3],[10,4]]

function min_max(data, feature_count) { 
    const cloned_data = _.cloneDeep(data)
    for (let i = 0; i < feature_count; i++) {
      const column = cloned_data.map(row => row[i])
      // console.log('column', column)
      const min = _.min(column)
      const max = _.max(column)
      // console.log(min,max)
      console.log(cloned_data)
      for (let j = 0; j < cloned_data.length; j++) {
        // console.log(cloned_data[j][i])
        cloned_data[j, i] = (cloned_data[j][i] - min) / (max - min)
        console.log(cloned_data[j, i])
      }
    }
    return cloned_data
  }

  console.log(min_max(data,1))
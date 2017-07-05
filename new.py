import numpy as np 
from scipy import stats

def dummy_data_random():
  return np.random.randn(100)*20

def dummy_data_linear():
  return np.array([i for i in range(100)])+np.random.randn(100)

def dummy_data_constant():
  return np.array([1 for _ in range(100)])

def dummy_data_jump():
  return np.array([ [i*10 for _ in range(20)] for i in range(5)]).reshape(-1)

def normalize(data):
  data = data.astype(float)
  min_value = min(data)
  max_value = max(data)
  gap = max_value - min_value
  if gap < 0.1:
    data = data - min_value
  else:
    data = (data - min_value)/(max_value - min_value)
  return data, max_value, min_value

def get_threshold(data):
  upperbound = -10000.0
  lowerbound = 10000.0
  for i in range(len(data)-1):
    upperbound = max(upperbound, data[i+1]-data[i])
    lowerbound = min(lowerbound, data[i+1]-data[i])
  return upperbound, lowerbound

def get_prediction(nextdata, data):
  data = data.astype(float)
  normalized_data, max_value, min_value = normalize(data)
  upperbound, lowerbound = get_threshold(normalized_data)
  upperbound = upperbound * (max_value-min_value)
  lowerbound = lowerbound * (max_value-min_value)
  print("The prediction is %.2f, the upperbound is %.2f, the lowerbound is %.2f"%
                             (nextdata, data[-1]+upperbound, data[-1]+lowerbound))
def check_natura_change(data):
  pre_slope,_,_, _,pre_std_err = stats.linregress(
                        [i for i in range(len(data[:-30]))], data[:-30])
  end_slope,_,_, _,end_std_err = stats.linregress(
                        [i for i in range(len(data[-30:]))], data[-30:])
  print("slope %.2f %.2f"%(pre_slope, end_slope))
  print("stderr %.2f %.2f"%(pre_std_err, end_std_err))
  if abs(end_slope-pre_slope)/abs(pre_slope+0.01) > 1 or abs(end_std_err - pre_std_err)/abs(pre_std_err+0.01) > 1:
    print("============Nature Change!!!!================")

def main():
  data = dummy_data_random()
  # Be careful, the data need to be float!!!!
  data = data.astype(float)
  normalized_data, max_value, min_value = normalize(data)
  upperbound, lowerbound = get_threshold(normalized_data)

  # Check whether the nature change
  check_natura_change(normalized_data)

  # Check whether the last 30 data is unchanged or not:
  _, truncated_max, truncated_min = normalize(data[-30:])
  if truncated_min == truncated_max:
    print("WARNING: The last 30 days' data nerver changed")
    return 

  # Check whether it is linear or not:
  gap = max_value - min_value
  if gap < 0.1:
    # If the data is almost constant, we don't need to do anything
    print("Data is almost constant, the constant number is %.2f"%(min_value,))
    return
  else:
    # Then check whether it is linear or not, checked by r_value:
    slope, intercept, r_value, _, _ = stats.linregress(
                        [i for i in range(len(normalized_data))], normalized_data)
    print("The slope is %.2f, The intercep is %.2f, The r value is %.2f"%
                                            (slope, intercept, r_value))
    if r_value**2 > 0.98:
      # The data is linear
      if slope < 0.0:
        print("According to the r_value square, the data is linear, and decreasing.")
      else:
        print("According to the r_value square, the data is linear, and increasing.")
      # Predict next data: using least square linear regression method:
      nextdata = (intercept + slope*len(data))*(max_value-min_value) + min_value
      get_prediction(nextdata, data)
    else:
      print("According to the r_value, the data is not linear."
         +  " Then use auto_regression to predict.")
      # TODO Use Auto regression to predict
      nextdata = (intercept + slope*len(data))*(max_value-min_value) + min_value
      get_prediction(nextdata, data)
      
    # get_prediction
      

if __name__ == '__main__':
  main()
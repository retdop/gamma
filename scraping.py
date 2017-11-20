import os
from selenium import webdriver
import time
from datetime import datetime, timedelta

def letsgo(driver):
    print('letsgo')
    uri = 'https://bitcoincharts.com/charts/krakenEUR'
    driver.get(uri)

    driver.find_element_by_id('c').click()
    driver.find_element_by_id('s').clear()
    driver.find_element_by_id('s').clear()
    driver.find_element_by_id('i').send_keys('5-min')
    driver.execute_script('load_table()')
    print('letsgo done')

def change_dates(driver, start_date, end_date):
    print('changing dates')
    driver.find_element_by_id('s').clear()
    driver.find_element_by_id('e').clear()
    print(start_date, end_date)
    driver.find_element_by_id('e').send_keys(start_date)
    driver.find_element_by_id('e').send_keys('\n')
    print(1)
    driver.find_element_by_id('s').send_keys(end_date)
    driver.find_element_by_id('s').send_keys('\n')
    print(2)
    time.sleep(1)
    #date = driver.find_element_by_class_name('ui-state-active')
    print(3)
    time.sleep(1)
    driver.execute_script('load_table()')
    print('dates changed')

def save_data(driver, start_date, end_date):
    print('saving data')
    filename = 'data/data_' + start_date + '-' + end_date + '.txt'
    time.sleep(1)
    yo = driver.find_element_by_id('chart_table').text
    print(len(yo))
    while (len(yo) < 100):
        yo = driver.find_element_by_id('chart_table').text
        print(len(yo))
    with open(filename, "w") as text_file:
        print(yo, file=text_file)
    print('data saved to file ' + filename)

if __name__=='__main__':

    os.chdir('/home/gabriel/fun/trading/gamma/')

    dates = ['2017-11-20', '2017-11-10', '2017-10-31', '2017-10-21', '2017-10-11', '2017-10-01']
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10)
    driver = webdriver.Firefox(executable_path="/opt/geckodriver")
    letsgo(driver)
    for i in range(16):
        print('NOW DOING ' + str(i) + ' OUT OF 16')
        print(start_date)
        change_dates(driver, end_date.strftime('20%y-%m-%d'), start_date.strftime('20%y-%m-%d'))
        save_data(driver, end_date.strftime('20%y-%m-%d'), start_date.strftime('20%y-%m-%d'))
        start_date = start_date - timedelta(days=10)
        end_date = end_date - timedelta(days=10)

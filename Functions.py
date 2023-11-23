# Databricks notebook source
import pandas as pd

# COMMAND ----------

def disp_all(df):

    max_rows = pd.get_option('display.max_rows')
    max_columns = pd.get_option('display.max_columns')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    display(df)

    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)

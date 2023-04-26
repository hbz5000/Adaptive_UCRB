# cm2015.ctl
#
# Control file for the Colorado monthly modeling calibration efforts for
#        1909-2013 simulations including soil moisture accounting
#
 Upper Colorado River Basin
 Historic Diversions
    1940     : iystr   STARTING YEAR OF SIMULATION
    2013     : iyend   ENDING YEAR OF SIMULATION
       2     : iresop  OUTPUT UNIT OPTION. 1 FOR [CFS], 2 FOR [AF], 3 FOR [KAF]
       0     : moneva  TYPE OF EVAP. DATA. 0 FOR VARIANT DATA. 1 FOR CONS. DATA
       1     : ipflo   TYPE OF STREAM INFLOW. 1 FOR TOTAL FLOW. 2 FOR GAINS
       0     : numpre  NO. OF PRECIPITATION STATIONS
      12     : numeva  NO. OF EVAPORATION STATIONS
      -1     : interv  NO. OF TIME INTERVALS IN DELAY TABLE. MAXIMUM=60.
  1.9835     : factor  FACTOR TO CONVERT CFS TO AC-FT/DAY (1.9835)
  1.9835     : rfacto  DIVISOR FOR STREAM FLOW DATA;    ENTER 0 FOR DATA IN cfs, ENTER 1.9835 FOR DATA IN af/mo
  1.9835     : dfacto  DIVISOR FOR DIVERSION DATA;      ENTER 0 FOR DATA IN cfs, ENTER 1.9835 FOR DATA IN af/mo
       0     : ffacto  DIVISOR FOR IN-STREAM FLOW DATA; ENTER 0 FOR DATA IN cfs, ENTER 1.9835 FOR DATA IN af/mo
  1.0        : cfacto  FACTOR TO CONVERT RESERVOIR CONTENT TO AC-FT
  1.0        : efacto  FACTOR TO CONVERT EVAPORATION DATA TO FEET
  1.0        : pfacto  FACTOR TO CONVERT PRECIPITATION DATA TO FEET
  WYR        : cyr1    Year type (a5, All caps, right justified !!)
       1     : icondem 1=no add; 2=add, 3=total demand in *.ddm
       0     : ichk    0 = off, 1=print river network,  -n= detailed printout at river node ichk
       1     : ireopx  Re-operation switch (0=re-operate;1=no re-operation)
       1     : ireach  Switch for instream flow reach approach (0=No insream flow reach approach, 1=Instream reach approach)
       1     : icall   Switch for detailed call data (0=no detailed call data; 1=yes detailed call data)
       0     : ccall   Detailed call water right ID (not used if icall=0)
       0     : iday    Switch for daily calculations (0=monthly analysis; 1=daily analysis)
       0     : iwell   Switch for well operations (0=no wells in *.rsp;-1=no wells but in *.rsp;1=yes wells no max limit;2 yes const max limit; 3 yes, variable max limit in *.rin
       0     : gwmaxrc Constant maximum recharge limit (cfs); only used when iwell = 2
       0     : isjrip  Switch for an annual San Juan Recovery Program Sediment file
      10     : itsfile Switch for an annual time series file (-1=no *.tsp but in *.rsp,0=no tsfile,1=RGDSS GW acres ts file, 2=SJRIP ts file)
       1     : ieffmax Switch for irrigation water requirement (IRW) file
       0     : isprink Switch for sprinkler data (area and efficiency) use (0=off, 1=Maximum Supply, 2=Mutual Supply
       3     : soild   Switch for soil moisture accounting

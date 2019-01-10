#include <cstddef>
#include <iostream>
#include <fstream>

//Threshhold and number of runs that evening. The date as well
TString threshhold = "100";
int NUM_RUNS = 10;
TString date = "jan_2_2019";



//MAC

TString READ_PATH = "/Users/lukepolson/Desktop/Jan_Test_Data/test2_alpha_data";
TString WRITE_PATH = "/Users/lukepolson/Documents/GitHub/School/Phys 499/datafiles/"+date+"/data";

  
// File where events are read from. Trees and file used in main method
TString filename;
TFile *f;
TTree *t;

double SM_P_NS=0.5; // Samples Per Nanosecond
double TRIGTIME=10000; 
int NUM_EVENTS =4100; // Number of Events Read From sample file

// The range to look for the trigger time when doing integral ratio stuff
double TRIGINIT = 5000; 
double TRIGFIN = 13000;

// This is the channel number used with the digitizer
TString c = "0";


// Helper method used to get the baseline average of a given event (set of data)
double getbaseline(vector<double> *data) {
  int datapoints=0;
  int sum=0;
  for(int i=0; i<TRIGINIT*SM_P_NS; i++){ 
    // Only loops through samples on baseline
    sum=sum+data->at(i);
    datapoints=datapoints+1;
  }

  // This is the baseline average
  return sum/datapoints;
}


// Returns value of integral of waveform at time 'timevalue' after the trigger point (microseconds)
double get_integral(vector<double> *data, int event, double timevalue) {
    
  t->GetEntry(event);

  double baselineavg = getbaseline(data);

  vector<double> WFintegral(data->size()); // Waveform integral
  double deltat = 1/SM_P_NS;
  double sum = 0;
  
  for(int i=0; i<data->size(); i++){ 
    WFintegral[i] = sum + (-((data->at(i))-baselineavg)); // Minus sign used b/c pulse is negative
    sum = sum + (-((data->at(i))-baselineavg));
  }

  // Finds Trigger Point
  //Starts off assuming trigger time is 10000 but does a search to see otherwise
  double trigtime = TRIGTIME;
  double max_energy = WFintegral.back(); //last element

  double time_ten_percent = 0;
  double time_twenty_percent = 0;
  double value_ten_percent = 0;
  double value_twenty_percent = 0;
  
  for(int i=0; i<data->size(); i++) {
    if(WFintegral[i]>max_energy/10) {
      value_ten_percent = WFintegral[i];
      time_ten_percent = i*deltat;
      break;
    }
  }

  for(int i=0; i<data->size(); i++) {
    if(WFintegral[i]>max_energy/20) {
      time_twenty_percent = WFintegral[i];
      time_twenty_percent = i*deltat;
      break;
    }
  }

  if(time_ten_percent != 0 && time_twenty_percent !=0 && time_ten_percent != time_twenty_percent) {
    trigtime = time_ten_percent+value_ten_percent*((time_twenty_percent-time_ten_percent)/(value_ten_percent-value_twenty_percent));
      }


  // Finds value at time 'timevalue'
  double val = WFintegral[(int)(trigtime/deltat+timevalue/deltat)];;
  return val;
}


double get_time(double tm, int event) {

  t->GetEntry(event);
  return tm;
  
}

bool passeventcuts(vector<double>& data, int event) {
  return true;
}

void raw_to_csv() {

  for(int i=1; i<=NUM_RUNS; i++) {
    //File that data is read from
    f = new TFile(READ_PATH+i+".root", "READ");
    t = (TTree*)f->Get("data");
  
    vector<double> *data = new vector<double>;
    double_t tm;
    t->SetBranchAddress("time", &tm);
    t->SetBranchAddress("ch"+c, &data); //Data now refers to that specific branch in the tree

    cout<<get_integral(data, 5, 1200) << endl;
    cout<<get_integral(data, 5, 7400) << endl;
    cout<<get_time(tm, 5) <<endl;

    std::ofstream myfile;
    myfile.open (WRITE_PATH+i+".csv");
    myfile << "Time,Integral 1200,Integral 7400\n";
    for(int j=0; j<t->GetEntries(); j++) {
      myfile << std::fixed << std::setprecision(6)<< get_time(tm, j);
      myfile << ",";
      myfile << std::fixed << std::setprecision(6)<< get_integral(data, j, 1200);
      myfile << ",";
      myfile << std::fixed << std::setprecision(6)<< get_integral(data, j, 7400);
      myfile << "\n";
    }
    myfile.close();
  
}
}


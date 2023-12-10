const int fsrPin1 = A0; // Analog input pin for FSR 1
const int fsrPin2 = A1; // Analog input pin for FSR 2

void setup() {
  Serial.begin(9600); // Initialize serial communication
}

void loop() {
  // Read analog values from FSRs
  int fsrValue1 = analogRead(fsrPin1);
  int fsrValue2 = analogRead(fsrPin2);

  // Send FSR 1 data to the Serial Plotter
  
  if(fsrValue1>450)
  {
    Serial.println(1);
//  Serial.write(1);
  }
  else
  {
    Serial.println(0);
//    Serial.write(0);
  }
  // Send FSR 2 data to the Serial Plotter
  delay(10);
//  Serial.print("FSR2:");
//  Serial.println(fsrValue2);
  if(fsrValue2>500)
  {
    Serial.println(fsrValue2);
//  Serial.write(1);
  }
  else
  {
    Serial.println(fsrValue2);
//    Serial.write(0);
  }
  delay(1000); // Adjust the delay as needed
}

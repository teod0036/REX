//                   version 1.1.1      Kim Steenstrup Pedersen, DIKU, August 2017, August 2020
//
//  This sketch converts an Arduino Uno with a BOE Shield and Arlo DBH-10 motor bridge controller 
//  into a serial robot controller. The sketch is inspired by Frindo serial_Robot.
//
//  The sketch provides basic motion control as well as access to Arduinos analog inputs over the
//  serial or USB port for robotics applications. The simple commands means that the controller can be
//  interfaced to anything with a serial or USB port including a PC, Raspberry Pi, microcontroller 
//  or another Arduino.
//
//  This sketch uses the ArloRobot library and provides
//  a mix of stepped commands, continuous commands and analog measurements. The movement of your robot 
//  is controlled by sending a single character command over the serial port.
//  
//  Stepped commands make your robot move in a specific way for a fixed period of time and speed. The length of 
//  time is set using the "step_time" and "turn_time" variables. All times are in ms (milliseconds). 
//  The step commands are:
//
//    f = Forward, b = Backwards, l = rotate Left, r = rotate Right 
//
//  Continuous commands make your robot move in the same way continuously until you send a stop or other command:  
//
//    s = Stop, g = Go, v = Reverse, n = rotate Left, m = rotate Right 
//
//  The speed of the motors during all movements is set by the "motor_speed" and "turn_speed". Motor speeds 
//  are measured in wheel encoder counts per seconds. There are 144 counts for a complete wheel revolution. 
//
//  Set the motor_speed and turn_speed with these commands followed by an integer representing the wanted speed.
//
//    z = motor_speed , x = turn_speed
//
//  Set the step_time used by f and b and the turn_time used by l and r followed by an integer representing the 
//  wanted time in miliseconds:
//
//    t = step_time, y = turn_time
//
//  Start motors with different speeds and directions
//
//    d = Go differential
//
//  followed by 4 integers separated by commas: powerLeft, powerRight, dirLeft (0 = reverse and 1 = forward) 
//  and dirRight (0 = reverse and 1 = forward). Example:
// 
//    d127,72,1,0
//
//  Make a distance measurement using the sonar ping sensors. The distance will be returned in mm.
//
//    0 = Front, 1 = Back, 2 = Left, 3 = Right
//
//  Read the left wheel encoder (there are 144 counts for a complete wheel revolution and this returns the counts
//  since last clear counts command) by
//
//    e0 = left wheel encoder count
//
//  and read the right wheel encoder by
//
//    e1 = right wheel encoder count
//
//  Clear the internal wheel encoder counts by
//
//    c = clear the encoder counts
//
//  Shutdown the DBH-10 motor bridge card and clean-up
//
//    k = shutdown



#include <ArloRobot.h>
#include <SoftwareSerial.h>

// Set wheel encoder pins
#define encoderLA 4  // Left A channel pin
#define encoderLB 2  // Left B channel pin (interrupt)
#define encoderRA 7  // Right A channel pin
#define encoderRB 3  // Right B channel pin (interrupt)


// Arlo and serial objects required 
ArloRobot Arlo;                               // Arlo object
SoftwareSerial ArloSerial(12, 13);            // Serial in I/O 12, out I/O 13


// define variables used
int Response;
int motor_speed = 32; // encoder counter / sec
int turn_speed = 32;  // encoder counter / sec
int step_time = 1000; //mSec
int turn_time = 1000; //mSec

volatile long left_encoder = 0; // Left wheel encoder counter
volatile long right_encoder = 0; // Right wheel encoder counter 


// Interrupt handlers
void runEncoderLeft()
{
 if (digitalRead(encoderLA) == digitalRead(encoderLB))   
   left_encoder++;    // count up if both encoder pins are HIGH on pin change interrupt
 else                                       
   left_encoder--;    //count down if pins are HIGH and LOW
}

void runEncoderRight()
{
 if (digitalRead(encoderRA) == digitalRead(encoderRB))   
   right_encoder--;    // count down if both encoder pins are HIGH on pin change interrupt
 else                                       
   right_encoder++;    //count up if pins are HIGH and LOW
}



void setup() {
  Serial.begin(9600);             // set up serial port
    
  ArloSerial.begin(19200);                    // Start DHB-10 serial com
  Arlo.begin(ArloSerial);                     // Pass to Arlo object

  // Wheel encoder pins
  pinMode(encoderLA, INPUT);
  pinMode(encoderLB, INPUT);
  pinMode(encoderRA, INPUT);
  pinMode(encoderRB, INPUT);

  // Set wheel encoder interrupt handlers
  attachInterrupt(digitalPinToInterrupt(encoderLB), runEncoderLeft, RISING); 
  attachInterrupt(digitalPinToInterrupt(encoderRB), runEncoderRight, RISING); 
}

void loop() {
 
    if (Serial.available() > 0)            // if something has been received 
      {
        int incoming = Serial.read();      // go read it
        
        if ((char)incoming == 'f')
        {
          stepMove(step_time, motor_speed, motor_speed);
          
          Serial.println("Forward");
        }
        
        else if ((char)incoming == 'b')
        {
          stepMove(step_time, -motor_speed, -motor_speed);
          Serial.println("Back");
        }
        
        else if ((char)incoming == 'l')
        {
          stepMove(turn_time, -turn_speed, turn_speed);
          Serial.println("Left");
        }
        
        else if ((char)incoming == 'r')
        {
          stepMove(turn_time, turn_speed, -turn_speed);
          Serial.println("Right");
        }
        
        else if ((char)incoming == 's')
        {          
          //Arlo.writeSpeeds(0,0); 
          Arlo.writeMotorPower(0,0);
          Serial.println("Stop");
        }
        
        else if ((char)incoming == 'g')
        {
          Arlo.writeSpeeds(motor_speed, motor_speed); 
          Serial.println("Go");
        }
        
        else if ((char)incoming == 'v')
        {
          Arlo.writeSpeeds(-motor_speed, -motor_speed); 
          Serial.println("Reverse");
        }
        
        else if ((char)incoming == 'm')
        {
          Arlo.writeSpeeds(turn_speed, -turn_speed);
          Serial.println("Clockwise");
        }
        
        else if ((char)incoming == 'n')
        {
          Arlo.writeSpeeds(-turn_speed, turn_speed);
          Serial.println("Counter Clockwise");
        }
        
        else if ((char)incoming == '0')
        {          
          //Response = pingCm(11); // Front sensor
          Response = pingMm(11); // Front sensor
          Serial.println(Response); 
        }

        else if ((char)incoming == '1')
        {
          //Response = pingCm(10); // Back sensor
          Response = pingMm(10); // Back sensor
          Serial.println(Response); 
        }
        
        else if ((char)incoming == '2')
        {
          //Response = pingCm(9); // Left sensor
          Response = pingMm(9); // Left sensor
          Serial.println(Response); 
        }
        
        else if ((char)incoming == '3')
        {
          //Response = pingCm(8); // Right sensor
          Response = pingMm(8); // Right sensor
          Serial.println(Response); 
        }

        else if ((char)incoming == 'z')
        {
          String inString = "";
          int len = 4;
          char inChars[len];
          Serial.readBytesUntil('\n', inChars, len);
          for (int i=0; i < len+1; i++) { 
            if (isDigit(inChars[i])) {
              inString += (char)inChars[i];
            }
          }
          motor_speed = inString.toInt();
          Serial.print("Setting motor speed to ");
          Serial.println(motor_speed);
        }
  
        else if ((char)incoming == 'x')
        {          
          String inString = "";
          int len = 4;
          char inChars[len];
          Serial.readBytesUntil('\n', inChars, len);
          for (int i=0; i < len+1; i++) { 
            if (isDigit(inChars[i])) {
              inString += (char)inChars[i];
            }
          }
          turn_speed = inString.toInt();
          Serial.print("Setting turn speed to ");
          Serial.println(turn_speed);
        }
        
        else if ((char)incoming == 't')
        {
          String inString = "";
          int len = 5;
          char inChars[len];
          Serial.readBytesUntil('\n', inChars, len);
          for (int i=0; i < len; i++) { 
            if (isDigit(inChars[i])) {
              inString += (char)inChars[i];
            }
          }
          step_time = inString.toInt();
          Serial.print("Setting step_time to ");
          Serial.println(step_time);
        }
        
        else if ((char)incoming == 'y')
        {
          String inString = "";
          int len = 5;
          char inChars[len];
          Serial.readBytesUntil('\n', inChars, len);
          for (int i=0; i < len; i++) { 
            if (isDigit(inChars[i])) {
              inString += (char)inChars[i];
            }
          }
          turn_time = inString.toInt();
          Serial.print("Setting turn_time to ");
          Serial.println(turn_time);
        }
        
        else if ((char)incoming == 'd')
        {
          int params[4];
          int idx = 0;
          String inString = "";
          
          int len = 13;
          char inChars[len];
          int numBytes = Serial.readBytesUntil('\n', inChars, len);
          for (int i=0; i < numBytes; i++) { 
            if (isDigit(inChars[i])) {
              inString += (char)inChars[i];
            } else if (inChars[i] ==',') {
              params[idx] = inString.toInt();
              inString = "";
              idx++;
            }
          }
          
          if ((char)inString[0] == 48) { // ASCII '0'=48
            params[idx] = 0;
          } else { // Assuming it is '1'
            params[idx] = 1;
          }
          
          
          idx++;
                    
          if (idx == 4) {
            int dirLeft = 1;
            if (params[2] == 0)
                dirLeft = -1;
            int dirRight = 1;
            if (params[3] == 0)
                dirRight = -1;
            
            //Arlo.writeSpeeds(dirLeft*params[0], dirRight*params[1]);
            
            Arlo.writeMotorPower(dirLeft*params[0], dirRight*params[1]);
            
            Serial.print("Go diff ");
            Serial.print(params[0]);
            Serial.print(",");
            Serial.print(params[1]);
            Serial.print(",");
            Serial.print(params[2]);
            Serial.print(",");
            Serial.println(params[3]);
          } else {
            Serial.print("ERROR: ");
            for (int i=0; i < len; i++) { 
              inString += (char)inChars[i];
            }
            Serial.println(inString);
          }
        }
        
        else if ((char)incoming == 'c')
        {
          // Clear the wheel encoder counts
          //Arlo.clearCounts();
          // Critical region
          noInterrupts();
          left_encoder = 0;
          right_encoder = 0;
          interrupts();
          Serial.println("Resetting counts");
        }
        
        else if ((char)incoming == 'e')
        {
          // Read the wheel encoder counts
          
          String inString = "";
          int len = 1;
          char inChars[len];
          Serial.readBytesUntil('\n', inChars, len);
          for (int i=0; i < len; i++) { 
            if (isDigit(inChars[i])) {
              inString += (char)inChars[i];
            }
          }
          
          if (inString.toInt() == 0) {
            //Serial.println(Arlo.readCountsLeft());
            noInterrupts();
            // Inside critical region
            long counts = left_encoder;
            interrupts();
            Serial.println(counts);
          } else if (inString.toInt() == 1) {
            //Serial.println(Arlo.readCountsRight());
            noInterrupts();
            // Inside critical region
            long counts = right_encoder;
            interrupts();
            Serial.println(counts);
          } else {
            Serial.println("ERROR - unknown wheel encoder");
          }
        }
        else if ((char)incoming == 'k')
        {
          Arlo.end();
          Serial.println("Shutdown");
        }
    }        

}

void stepMove(unsigned int dT, int speedLeft, int speedRight) {
  Arlo.writeSpeeds(speedLeft, speedRight);
  delay(dT);
  Arlo.writeSpeeds(0, 0);
}


int pingCm(int pin)                           // Ping measurement function
{
  digitalWrite(pin, LOW);                     // Pin to output-low
  pinMode(pin, OUTPUT);                       
  delayMicroseconds(200);                     // Required between successive
  digitalWrite(pin, HIGH);                    // Send high pulse
  delayMicroseconds(5);                       // Must be at least 2 us
  digitalWrite(pin, LOW);                     // End pulse to start ping
  pinMode(pin, INPUT);                        // Change to input
  long microseconds = pulseIn(pin, HIGH);     // Wait for echo to reflect
  return microseconds / 29 / 2;               // Convert us echo to cm
}

int pingMm(int pin)                           // Ping measurement function
{
  digitalWrite(pin, LOW);                     // Pin to output-low
  pinMode(pin, OUTPUT);                       
  delayMicroseconds(200);                     // Required between successive
  digitalWrite(pin, HIGH);                    // Send high pulse
  delayMicroseconds(5);                       // Must be at least 2 us
  digitalWrite(pin, LOW);                     // End pulse to start ping
  pinMode(pin, INPUT);                        // Change to input
  long microseconds = pulseIn(pin, HIGH);     // Wait for echo to reflect
  float fmicrosecs = microseconds / 29.0f / 2.0f * 10.0f;
  return (int)fmicrosecs;                     // Convert US echo to mm
}

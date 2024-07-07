/*
	This program uses randomly generated numbers to give a fictional golf swing 
	calculation. This was created to be a fun side project to compliment my classwork.
*/
import java.util.Scanner;
import java.lang.Math;
public class GolfSwingGenerator {
	public static void main(String[] args){
		int initialRange = initialRange();
		double angleCalculation = (angleCalculation()) ;
		double cosAngle = Math.cos(angleCalculation);
		double tanAngle = Math.tan(angleCalculation);
		System.out.println(cosAngle);
		System.out.println(tanAngle);
		double howFarBallWent = strikeDistanceCalculation();

		double strikeDistanceCalculation = howFarBallWent*cosAngle;
		
		double yardsToHole = initialRange-(int)strikeDistanceCalculation;
		System.out.println(angleCalculation+" club strike angle");
		System.out.println(initialRange+" Initial Range to the hole");
		System.out.println(strikeDistanceCalculation+" hit distance in yards");
		System.out.println(yardsToHole+" Yards left to the hole");
	}

	public static int initialRange(){
		int[] holeYardage = {150,200,250,300,350,400,500,550,600};
		int randomHoleYardage = (int) (Math.random()*holeYardage.length);
		int setHoleYardage = holeYardage[randomHoleYardage];
		return setHoleYardage;	
	}

	public static double angleCalculation(){
		//Returns a random angle from the given array
		int[] angle = {1,20,40,80,34,67,23,89};
		int randomHitAngle = (int) (Math.random()*angle.length);
		int hitAngle = angle[randomHitAngle];		
		
		return hitAngle;
	}
	public static int strikeDistanceCalculation(){
		int[] power = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
		int randomPower = (int) (Math.random()*power.length);
		int swingPower = power[randomPower];
		int hitDistance = swingPower*3;
		
		System.out.println(swingPower+"%");
		System.out.println(hitDistance+" force");
		return hitDistance;
	}

}
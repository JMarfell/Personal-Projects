import java.util.Scanner;

public class celsiusCalculator{
	public static void main (String[] args){
		System.out.println("Please enter a tempuature in celsius.");
		Scanner scanIn = new Scanner(System.in);
		double TempInCelsius = scanIn.nextDouble();
		double fahrenheit = (TempInCelsius*1.8)+32;
		System.out.println("The tempuature in fahrenheit is : "+fahrenheit);
	}

}
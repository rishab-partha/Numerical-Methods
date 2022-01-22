/**
 * The FloatingPoint class investigates a number of features of floating point arithmetic in Java,
 * such as the truncation of doubles and the calculation of double maximum, minimum, and epsilon
 * values. This class also defines a main method that tests various operations with derived values
 * of zero and infinity, comparing results derived from computational methods against the official
 * Java database.
 * 
 * @author Rishab Parthasarathy
 * @version 01.22.2022
 */
public class FloatingPoint
{
    static final double INF = 1.0/0.0;
    static final double NEG_INF = -1.0*INF;

    /**
     * Method main tests first tests these 7 conditions of floating point arithmetic in Java,
     * evaluating the results produced by:
     *      1. nonzero / zero
     *      2. zero / zero
     *      3. +- Infinity / zero
     *      4. zero / +- Infinity
     *      5. zero * +- Infinity
     *      6. +- Infinity * +- Infinity
     *      7. +- Infinity / +- Infinity
     * 
     * where Infinity is calculated using the results of the nonzero / zero operation on
     * doubles. Then, the main method calls functions to calculate +- MAX, +- MIN, and 
     * +- EPS, comparing the results derived against those stored in the Java Double and Math
     * classes.
     * 
     * @param args a String array of command line arguments
     */
    public static void main(String[] args)
    {
        FloatingPoint tester = new FloatingPoint();

        // nonzero (int) / zero (int)
        try
        {
            System.out.println(1/0);
        }
        catch (Exception e)
        {
            System.out.println("nonzero (int) / zero (int): " + 
                    "Throws an Arithmetic Exception: / by zero");
        }

        // nonzero (double) / zero (double)
        System.out.println("nonzero (double) / zero (double): " + 1.0/0.0 + "\n");

        // zero (int) / zero (int)
        try
        {
            System.out.println(0/0);
        }
        catch (Exception e)
        {
            System.out.println("zero (int) / zero (int): " + 
                    "Throws an Arithmetic Exception: / by zero");
        }

        // zero (double) / zero (double)
        System.out.println("zero (double) / zero (double): " + 0.0/0.0 + "\n");

        // infinity (double) / zero (double)
        System.out.println("infinity (double) / zero (double): " + INF/0.0);

        // -infinity (double) / zero (double)
        System.out.println("-infinity (double) / zero (double): " + NEG_INF/0.0 + "\n");

        // zero (double) / infinity (double)
        System.out.println("zero (double) / infinity (double): " + 0.0/INF);

        // zero (double) / -infinity (double)
        System.out.println("zero (double) / -infinity (double): " + 0.0/NEG_INF + "\n");

        // zero (double) * infinity (double)
        System.out.println("zero (double) * infinity (double): " + 0.0*INF);

        // zero (double) * -infinity (double)
        System.out.println("zero (double) * -infinity (double): " + 0.0*NEG_INF + "\n");

        // infinity (double) * infinity (double)
        System.out.println("infinity (double) * infinity (double): " + INF*INF);

        // infinity (double) * -infinity (double)
        System.out.println("infinity (double) * -infinity (double): " + INF*NEG_INF);

        // -infinity (double) * infinity (double)
        System.out.println("-infinity (double) * infinity (double): " + NEG_INF*INF);

        // -infinity (double) / -infinity (double)
        System.out.println("-infinity (double) / -infinity (double): " + NEG_INF*NEG_INF + "\n");

        // infinity (double) / infinity (double)
        System.out.println("infinity (double) / infinity (double): " + INF/INF);

        // infinity (double) / -infinity (double)
        System.out.println("infinity (double) / -infinity (double): " + INF/NEG_INF);

        // -infinity (double) / infinity (double)
        System.out.println("-infinity (double) / infinity (double): " + NEG_INF/INF);

        // -infinity (double) / -infinity (double)
        System.out.println("-infinity (double) / -infinity (double): " + NEG_INF/NEG_INF + "\n");

        // value of +MIN
        double min = tester.calcMin(1.0);
        System.out.println("Calculated value of +MIN: " + min);

        // value of -MIN
        double negMin = tester.calcMin(-1.0);
        System.out.println("Calculated value of -MIN: " + negMin);

        // Java stipulations on MIN
        System.out.println("Official JAVA value of MIN: " + Double.MIN_VALUE);
        System.out.println("Official JAVA value of MIN_EXP: " + Double.MIN_EXPONENT);
        System.out.println("Official JAVA value of MIN_NORMAL: " + Double.MIN_NORMAL + "\n");

        // value of +MAX
        double max = tester.calcMax(1.0);
        System.out.println("Calculated value of +MAX: " + max);

        // value of -MAX
        double negMax = tester.calcMax(-1.0);
        System.out.println("Calculated value of -MAX: " + negMax);

        // Java stipulations on MAX
        System.out.println("Official JAVA value of MAX: " + Double.MAX_VALUE);
        System.out.println("Official JAVA value of MAX_EXP: " + Double.MAX_EXPONENT + "\n");

        // value of +EPS
        double eps = tester.calcEps(1.0);
        System.out.println("Calculated value of +EPS: " + eps);

        // value of -EPS
        double negEps = tester.calcEps(-1.0);
        System.out.println("Calculated value of -EPS: " + negEps);

        // Java stipulation on EPS
        System.out.println("Official JAVA value of EPS: " + Math.ulp(1.0));
        return;
    }

    /**
     * Method calcMin uses binary descent to calculate the double MIN value in Java given some
     * starting value for the descent. Specifically, this method utilizes the formula:
     * 
     *      2.0 * (MIN / 2.0) != 0.0
     * 
     * Whenever this formula is true, calcMin tests the next value of MIN as half of the current
     * value, continuing in a binary descent until the smallest possible value of MIN has been
     * derived.
     * 
     * @param startingValue the starting point for the binary descent
     * @return the double MIN value calculated using the given binary descent
     */
    private double calcMin(double startingValue)
    {
        double curMin = startingValue;
        while (! (2.0*(curMin/2.0) == 0.0))
        {
            curMin /= 2.0;
        }
        curMin *= 2.0;
        return curMin;
    }

    /**
     * Method calcMax uses binary ascent to calculate the double MAX value in Java given some
     * starting value for the ascent. Specifically, this method utilizes the formula:
     * 
     *      (MAX * 2.0) / 2.0 != Infinity
     * 
     * Whenever this formula is true, calcMax tests the next value of MAX as double the current
     * value, continuing in a binary ascent until the largest possible value of MAX has been
     * derived.
     * 
     * @param startingValue the starting point for the binary ascent
     * @return the double MAX value calculated using the given binary ascent
     */
    private double calcMax(double startingValue)
    {
        double curMax = startingValue;
        while (! (((2.0*curMax)/2.0 == INF) || ((2.0*curMax)/2.0 == -1.0*INF)))
        {
            curMax *= 2.0;
        }
        curMax /= 2.0;
        return curMax;
    }

    /**
     * Method calcEps uses binary descent to calculate the double EPS value in Java given some
     * starting value for the descent. Specifically, this method utilizes the formula:
     * 
     *      1.0 + (EPS / 2.0) = 1.0 for EPS != 0
     * 
     * Whenever this formula is false, calcEps tests the next value of EPS as half of the current
     * value, continuing in a binary descent until the largest possible value of EPS has been
     * derived.
     * 
     * @param startingValue the starting point for the binary descent
     * @return the double EPS value calculated using the given binary descent
     */
    private double calcEps(double startingValue)
    {
        double curEps = startingValue;
        while ((1.0 + curEps/2.0) != 1.0)
        {
            curEps /= 2.0;
        }
        return curEps;
    }
}
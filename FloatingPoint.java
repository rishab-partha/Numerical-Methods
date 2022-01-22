import java.io.*;
import java.util.*;

public class FloatingPoint
{
    public static void main(String[] args)
    {
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
        System.out.println("nonzero (double) / zero (double): " + 1.0/0.0);
        final double inf = 1.0/0.0;
        final double negInf = -1.0*inf;

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
        System.out.println("zero (double) / zero (double): " + 0.0/0.0);

        // infinity (double) / zero (double)
        System.out.println("infinity (double) / zero (double): " + inf/0.0);

        // -infinity (double) / zero (double)
        System.out.println("-infinity (double) / zero (double): " + negInf/0.0);

        // zero (double) / infinity (double)
        System.out.println("zero (double) / infinity (double): " + 0.0/inf);

        // zero (double) / -infinity (double)
        System.out.println("zero (double) / -infinity (double): " + 0.0/negInf);

        // zero (double) * infinity (double)
        System.out.println("zero (double) * infinity (double): " + 0.0*inf);

        // zero (double) * -infinity (double)
        System.out.println("zero (double) * -infinity (double): " + 0.0*negInf);

        // infinity (double) * infinity (double)
        System.out.println("infinity (double) * infinity (double): " + inf*inf);

        // infinity (double) * -infinity (double)
        System.out.println("infinity (double) * -infinity (double): " + inf*negInf);

        // -infinity (double) * infinity (double)
        System.out.println("-infinity (double) * infinity (double): " + negInf*inf);

        // -infinity (double) / -infinity (double)
        System.out.println("-infinity (double) / -infinity (double): " + negInf*negInf);

        // infinity (double) / infinity (double)
        System.out.println("infinity (double) / infinity (double): " + inf/inf);

        // infinity (double) / -infinity (double)
        System.out.println("infinity (double) / -infinity (double): " + inf/negInf);

        // -infinity (double) / infinity (double)
        System.out.println("-infinity (double) / infinity (double): " + negInf/inf);

        // -infinity (double) / -infinity (double)
        System.out.println("-infinity (double) / -infinity (double): " + negInf/negInf);
    }
}
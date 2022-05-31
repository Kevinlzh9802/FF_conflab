function[angle] = get_angle(vector)
    x = vector(1);
    y = vector(2);
    angle = 0;
    
    if ((x==0) && (y==0))
%         disp(x)
%         disp(y)
        disp('error: 0 vector found')
    elseif(y==0)
        if (x>0)
            angle = 0;
        else
            angle = 180;
        end
    elseif(x==0)
        if(y>0)
            angle = 90;
        else
            angle = 270;
        end
    else
        angle0 = atan(abs(y/x));
        angle0 = angle0*180/pi;
        if ((x>0) && (y>0))
            angle = angle0;
        elseif ((x<0) && (y>0))
            angle = 180-angle0;
        elseif((x<0) && (y<0))
            angle = 180+angle0;
        else
            angle = 360-angle0;
        end
    end
    
    if(angle==360)
        angle=0;
    end
    
    if((angle<0)||(angle>360))
        error('Wrong angle found')
    end
end

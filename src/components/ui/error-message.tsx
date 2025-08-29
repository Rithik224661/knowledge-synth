import * as React from "react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ErrorMessageProps {
  title?: string;
  message: string;
  className?: string;
  variant?: "default" | "destructive";
  icon?: React.ReactNode;
  onRetry?: () => void;
  details?: string;
}

export function ErrorMessage({
  title = "Error",
  message,
  className,
  variant = "destructive",
  icon = <XCircle className="h-4 w-4" />,
  onRetry,
  details,
}: ErrorMessageProps) {
  const [showDetails, setShowDetails] = React.useState(false);

  return (
    <Alert variant={variant} className={cn("flex flex-col items-start gap-2", className)}>
      <div className="flex w-full items-start gap-2">
        {icon && <div className="mt-0.5">{icon}</div>}
        <div className="flex-1">
          <AlertTitle className="mb-1">{title}</AlertTitle>
          <AlertDescription className="text-sm">{message}</AlertDescription>
          
          {details && (
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs text-muted-foreground hover:underline mt-2"
            >
              {showDetails ? "Hide details" : "Show details"}
            </button>
          )}
          
          {showDetails && details && (
            <pre className="mt-2 p-2 bg-muted/50 rounded text-xs overflow-auto max-h-[200px] w-full">
              {details}
            </pre>
          )}
        </div>
      </div>
      
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 text-sm font-medium hover:underline self-end"
        >
          Try again
        </button>
      )}
    </Alert>
  );
}
import { Alert, AlertDescription } from "./ui/alert";
import { CheckCircle, AlertCircle, Info, XCircle } from "lucide-react";

export type AlertType = "success" | "error" | "warning" | "info";

interface StatusAlertProps {
  type: AlertType;
  message: string;
  onClose?: () => void;
}

export function StatusAlert({ type, message, onClose }: StatusAlertProps) {
  const alertStyles = {
    success: "border-green-500/50 bg-green-500/10 text-green-400",
    error: "border-red-500/50 bg-red-500/10 text-red-400",
    warning: "border-yellow-500/50 bg-yellow-500/10 text-yellow-400",
    info: "border-blue-500/50 bg-blue-500/10 text-blue-400",
  };

  const icons = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertCircle,
    info: Info,
  };

  const Icon = icons[type];

  return (
    <Alert className={`${alertStyles[type]} border relative`}>
      <Icon className="h-4 w-4" />
      <AlertDescription className="ml-2">{message}</AlertDescription>
      {onClose && (
        <button
          onClick={onClose}
          className="absolute top-2 right-2 opacity-70 hover:opacity-100 transition-opacity"
        >
          <XCircle className="h-4 w-4" />
        </button>
      )}
    </Alert>
  );
}

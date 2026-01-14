import { useState, useRef } from "react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { FileText, Upload, X, CheckCircle, AlertCircle } from "lucide-react";

const API_BASE_URL = "http://localhost:8000";

interface PdfUploadProps {
  onUploadSuccess?: (filename: string) => void;
  onUploadError?: (error: string) => void;
}

export function PdfUpload({ 
  onUploadSuccess,
  onUploadError,
}: PdfUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus("idle");
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.ok) {
        const data = await response.json();
        setUploadStatus("success");
        setStatusMessage(data.message);
        onUploadSuccess?.(file.name);
      } else {
        const error = await response.json();
        setUploadStatus("error");
        setStatusMessage(error.detail || "Upload failed");
        onUploadError?.(error.detail || "Upload failed");
      }
    } catch (error) {
      setUploadStatus("error");
      setStatusMessage("Failed to connect to server");
      onUploadError?.("Failed to connect to server");
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setSelectedFile(file);
      uploadFile(file);
    } else if (file) {
      alert("Please select a PDF file");
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadStatus("idle");
    setStatusMessage("");
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="space-y-2">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        className="hidden"
      />

      {!selectedFile ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleButtonClick}
          className="border-border hover:bg-accent text-foreground"
        >
          <Upload className="h-4 w-4 mr-2" />
          Upload PDF
        </Button>
      ) : (
        <div className="bg-muted/50 border border-border rounded-lg p-3 space-y-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-start gap-2 flex-1 min-w-0">
              {uploadStatus === "success" ? (
                <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
              ) : uploadStatus === "error" ? (
                <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
              ) : (
                <FileText className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatFileSize(selectedFile.size)}
                </p>
                {statusMessage && (
                  <p className={`text-xs mt-1 ${uploadStatus === "success" ? "text-green-500" : uploadStatus === "error" ? "text-red-500" : "text-muted-foreground"}`}>
                    {statusMessage}
                  </p>
                )}
              </div>
            </div>
            {!isUploading && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleRemoveFile}
                className="h-6 w-6 p-0 hover:bg-accent hover:text-destructive"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>

          {isUploading && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-1" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
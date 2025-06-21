import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { SquarePen, Brain, Send, StopCircle, Zap, Cpu } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ModelOption {
  id: string;
  name: string;
  provider: string;
}

interface ModelsResponse {
  models: ModelOption[];
  current_provider: string;
  default_models: {
    query_generator: string;
    reflection: string;
    answer: string;
  };
}

// Updated InputFormProps
interface InputFormProps {
  onSubmit: (inputValue: string, effort: string, model: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
}

export const InputForm: React.FC<InputFormProps> = ({
  onSubmit,
  onCancel,
  isLoading,
  hasHistory,
}) => {
  const [internalInputValue, setInternalInputValue] = useState("");
  const [effort, setEffort] = useState("medium");
  const [model, setModel] = useState("");
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:2024/api/models');
        if (response.ok) {
          const data: ModelsResponse = await response.json();
          setAvailableModels(data.models);
          // Set default model if available
          if (data.models.length > 0 && !model) {
            setModel(data.models[0].id);
          }
        } else {
          console.error('Failed to fetch models');
          // Fallback to hardcoded models if API fails
          setAvailableModels([
            {id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", provider: "google"},
            {id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", provider: "google"},
            {id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "google"},
          ]);
          setModel("gemini-2.0-flash");
        }
      } catch (error) {
        console.error('Error fetching models:', error);
        // Fallback to hardcoded models if fetch fails
        setAvailableModels([
          {id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", provider: "google"},
          {id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", provider: "google"},
          {id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "google"},
        ]);
        setModel("gemini-2.0-flash");
      } finally {
        setIsLoadingModels(false);
      }
    };

    fetchModels();
  }, []);

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue, effort, model);
    setInternalInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit with Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac)
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const isSubmitDisabled = !internalInputValue.trim() || isLoading;

  return (
    <form
      onSubmit={handleInternalSubmit}
      className={`flex flex-col gap-2 p-3 pb-4`}
    >
      <div
        className={`flex flex-row items-center justify-between text-white rounded-3xl rounded-bl-sm ${
          hasHistory ? "rounded-br-sm" : ""
        } break-words min-h-7 bg-neutral-700 px-4 pt-3 `}
      >
        <Textarea
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Who won the Euro 2024 and scored the most goals?"
          className={`w-full text-neutral-100 placeholder-neutral-500 resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none
                        md:text-base  min-h-[56px] max-h-[200px]`}
          rows={1}
        />
        <div className="-mt-3">
          {isLoading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 cursor-pointer rounded-full transition-all duration-200"
              onClick={onCancel}
            >
              <StopCircle className="h-5 w-5" />
            </Button>
          ) : (
            <Button
              type="submit"
              variant="ghost"
              className={`${
                isSubmitDisabled
                  ? "text-neutral-500"
                  : "text-blue-500 hover:text-blue-400 hover:bg-blue-500/10"
              } p-2 cursor-pointer rounded-full transition-all duration-200 text-base`}
              disabled={isSubmitDisabled}
            >
              Search
              <Send className="h-5 w-5" />
            </Button>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between">
        <div className="flex flex-row gap-2">
          <div className="flex flex-row gap-2 bg-neutral-700 border-neutral-600 text-neutral-300 focus:ring-neutral-500 rounded-xl rounded-t-sm pl-2  max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm">
              <Brain className="h-4 w-4 mr-2" />
              Effort
            </div>
            <Select value={effort} onValueChange={setEffort}>
              <SelectTrigger className="w-[120px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder="Effort" />
              </SelectTrigger>
              <SelectContent className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer">
                <SelectItem
                  value="low"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  Low
                </SelectItem>
                <SelectItem
                  value="medium"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  Medium
                </SelectItem>
                <SelectItem
                  value="high"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  High
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-row gap-2 bg-neutral-700 border-neutral-600 text-neutral-300 focus:ring-neutral-500 rounded-xl rounded-t-sm pl-2  max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm ml-2">
              <Cpu className="h-4 w-4 mr-2" />
              Model
            </div>
            <Select value={model} onValueChange={setModel} disabled={isLoadingModels}>
              <SelectTrigger className="w-[150px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder={isLoadingModels ? "Loading..." : "Model"} />
              </SelectTrigger>
              <SelectContent className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer">
                {availableModels.map((modelOption) => (
                  <SelectItem
                    key={modelOption.id}
                    value={modelOption.id}
                    className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                  >
                    <div className="flex items-center">
                      {modelOption.provider === 'google' && modelOption.name.includes('2.0') && (
                        <Zap className="h-4 w-4 mr-2 text-yellow-400" />
                      )}
                      {modelOption.provider === 'google' && modelOption.name.includes('2.5 Flash') && (
                        <Zap className="h-4 w-4 mr-2 text-orange-400" />
                      )}
                      {modelOption.provider === 'google' && modelOption.name.includes('Pro') && (
                        <Cpu className="h-4 w-4 mr-2 text-purple-400" />
                      )}
                      {modelOption.provider === 'openai' && (
                        <Cpu className="h-4 w-4 mr-2 text-green-400" />
                      )}
                      {modelOption.provider === 'openai_compatible' && (
                        <Cpu className="h-4 w-4 mr-2 text-blue-400" />
                      )}
                      {modelOption.name}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        {hasHistory && (
          <Button
            className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer rounded-xl rounded-t-sm pl-2 "
            variant="default"
            onClick={() => window.location.reload()}
          >
            <SquarePen size={16} />
            New Search
          </Button>
        )}
      </div>
    </form>
  );
};
